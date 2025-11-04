# debug
import debugpy
import socket
import logging

def setup_remote_debugging(port: int = 5678):

    ip = socket.gethostbyname(socket.gethostname())
    logging.info(f"Initializing remote debugging at {ip}:{port}")
    print(f"Initializing remote debugging at {ip}:{port}")

    debugpy.listen(address=(ip, port))
    
    debugpy.wait_for_client()    
    logging.info(f"Debugger attached: {debugpy.is_client_connected()}")

setup_remote_debugging()