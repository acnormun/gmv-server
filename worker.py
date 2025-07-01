from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

if __name__ == '__main__':
    print("🚀 Iniciando SimpleWorker da fila 'triagem' (Windows safe)")
    
    redis_conn = Redis()
    fila = Queue('triagem', connection=redis_conn)
    
    worker = SimpleWorker([fila], connection=redis_conn)
    worker.work()
