#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/var/www/car-detection/webapp/cardetector/')
sys.path.insert(0, '/var/www/car-detection/')

from cardetector.appserver import app as application
