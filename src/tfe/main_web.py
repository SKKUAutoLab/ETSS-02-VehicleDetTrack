"""Browser GUI of the vehicle detection & tracking pipeline.

Usage: python main_web.py [--host 127.0.0.1] [--allowed-host HOST] [--port 8000]
                          [--stream-fps N] [--root-dir DIR] [--fake-runner] [--fake-load-s S]
Then open http://127.0.0.1:8000. For other machines on the LAN, add --host 0.0.0.0
--allowed-host <this machine's LAN IP> and open http://<LAN IP>:8000 there; anyone who
reaches the port can use the GUI (no authentication). Never run it next to main.py:
they write the same result files.
"""
import argparse
import os
import shutil
import tempfile
from timeit import default_timer as timer

import uvicorn
from loguru import logger
from tfe.web.app import create_app

LOCAL_HOSTS = ("127.0.0.1", "localhost")  # Host headers from this machine and SSH tunnels; always accepted

# MARK: - Args

def parse_args(argv=None):
	parser = argparse.ArgumentParser(description="Browser GUI for vehicle detection & tracking")
	parser.add_argument("--host", default="127.0.0.1", help="Bind address; 0.0.0.0 also listens on the LAN, where anyone can use the GUI (no authentication).")
	parser.add_argument("--allowed-host", action="append", default=[], metavar="HOST",
	                    help="Also accept this Host header, e.g. this machine's LAN IP; repeatable. 127.0.0.1 and localhost are always accepted.")
	parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
	parser.add_argument("--stream-fps", type=float, default=None, help="Cap the video stream rate; default: pipeline FPS.")
	parser.add_argument("--root-dir", default=None, help="Root that contains data/; default: $ROOT_DIR or src/tfe.")
	parser.add_argument("--fake-runner", action="store_true", help="Synthetic runs without GPU or model, on a temporary copy of the data root.")
	parser.add_argument("--fake-load-s", type=float, default=0.2, help="Simulated model load time of --fake-runner.")
	return parser.parse_args(argv)

# MARK: - Server

class TFEServer(uvicorn.Server):
	"""``uvicorn.Server`` whose shutdown first ends the streams and asks the job to stop,
	because uvicorn waits for open connections before the lifespan shutdown (R8)."""

	def __init__(self, config: uvicorn.Config, app):
		super().__init__(config)
		self.tfe_app = app

	async def shutdown(self, sockets=None):
		self.tfe_app.state.begin_shutdown()
		await super().shutdown(sockets=sockets)


# MARK: - Main

def main(argv=None):
	process_start_time = timer()
	args               = parse_args(argv)
	os.environ["MPLBACKEND"] = "Agg"  # no Qt matplotlib backend on the server

	from tfe.web.paths import (
		resolve_root_dir,  # the ROOT_DIR rule without importing tfe.configuration (torch)
	)

	fake_root = None
	if args.fake_runner:
		from tfe.web.fake_runner import fake_runner_factory, make_fake_root
		if args.root_dir:
			root_dir = os.path.abspath(args.root_dir)
		else:  # fake runs never touch the real (gitignored) results (R11)
			fake_root = tempfile.mkdtemp(prefix="tfe-fake-root-")
			root_dir  = make_fake_root(resolve_root_dir(), fake_root)
		runner_factory = fake_runner_factory(load_s=args.fake_load_s)
	else:
		from tfe.web.session import camera_runner_factory
		root_dir       = os.path.abspath(args.root_dir or resolve_root_dir())
		runner_factory = camera_runner_factory

	os.environ["ROOT_DIR"] = root_dir  # read once by tfe.configuration, imported later in the job thread
	os.chdir(root_dir)                 # the YOLOX "variant" path in the configs is cwd-relative (R10)

	LAN_HOSTS = tuple(host for host in args.allowed_host if host not in LOCAL_HOSTS)

	app    = create_app(root_dir, runner_factory, stream_fps=args.stream_fps, allowed_hosts=LOCAL_HOSTS + LAN_HOSTS)
	config = uvicorn.Config(app, host=args.host, port=args.port, workers=1, reload=False, timeout_graceful_shutdown=5)
	server = TFEServer(config, app)
	logger.info(f"Web GUI: http://{args.host}:{args.port} (root {root_dir}{', fake runner' if args.fake_runner else ''})")
	if args.host not in LOCAL_HOSTS:  # other machines can reach it, and there is no password
		lan_urls = ", ".join(f"http://{host}:{args.port}" for host in args.allowed_host) or "none until --allowed-host <LAN IP> is given"
		logger.warning(f"LAN access without authentication; LAN URLs: {lan_urls}")
	try:
		server.run()
	except KeyboardInterrupt:  # uvicorn re-raises the captured Ctrl-C after its shutdown
		pass
	finally:
		if fake_root is not None:
			shutil.rmtree(fake_root, ignore_errors=True)
		total_process_time = timer() - process_start_time
		logger.success(f"Total processing time: {total_process_time} seconds.")


if __name__ == "__main__":
	main()
