import argparse
import asyncio
import logging

from ragroute.config import DATA_SOURCES, SUPPORTED_MODELS
from ragroute.ragroute import RAGRoute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="RAGRoute")
    parser.add_argument("--dataset", type=str, default="wikipedia", choices=["medrag", "feb4rag", "wikipedia"], help="The dataset being evaluated (influences the data sources)")
    parser.add_argument("--routing", type=str, default="ragroute", choices=["ragroute", "all", "random", "none"], help="The routing method to use - for random, we randomly pick n/2 of the n data sources")
    parser.add_argument("--disable-llm", action="store_true", help="Disable the LLM for testing purposes")
    parser.add_argument("--simulate", action="store_true", help="Simulate the system (for testing purposes)")
    parser.add_argument("--model", type=str, default=SUPPORTED_MODELS[0], choices=SUPPORTED_MODELS, help="The model to use for the LLM")
    args = parser.parse_args()
    
    ragroute = RAGRoute(args)
    try:
        asyncio.run(ragroute.start())
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        pass
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    
    logger.info("Exiting application")

if __name__ == "__main__":
    main()
