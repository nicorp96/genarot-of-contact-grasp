import argparse
import logging
from src.main import MainClass

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="GENERATOR OF CONTACT GRASP --MASTERARBEIT--NICOLAS-RODRIGUEZ"
        )
        parser.add_argument(
            "-c",
            "--config",
            help="path to a json config file",
        )
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG)
        main_program = MainClass(
            path=args.config,
            log_level=logging.DEBUG,
        )
        main_program.run()
    except Exception:
        logging.exception("Exception occured")
    finally:
        logging.debug("Program was succesfull finilized")
