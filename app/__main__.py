import argparse
from . import create_app

parser=argparse.ArgumentParser(description='Run the ECS171 local inference demo.')
parser.add_argument('--port',type=int,default=5172)
args=parser.parse_args()
if not 1024<=args.port<=65535:
    parser.error('port must be between 1024 and 65535')
create_app().run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False,threaded=True)
