import logging 

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(lineno)03d - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(name='[tracking]')

if __name__ == '__main__':
    logger.info('log initialized')