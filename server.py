import socketio
from main import recogApp
from waitress import serve
import socket

socketIO = socketio.Server()
appServer = socketio.WSGIApp(socketIO, recogApp)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

if __name__ == '__main__':
    serve(appServer, host=IPAddr, port=8080, url_scheme='http', threads=4, log_untrusted_proxy_headers=True)