import argparse
from types import SimpleNamespace

def parse_args():
   parser = argparse.ArgumentParser(prog='QITE')

   # Model settings
   model = parser.add_argument_group('model', description = "model name, norm type, encoder, etc.")
   model.add_argument('--model_name', type=str, default='QITE', help="{'QITE'}")
   model.add_argument('--num_qubits', type=int, default=1)
   
   # Task settings
   task = parser.add_argument_group('task', description = "task settings")
   task.add_argument('--task', type=str, default='demo', help="{'demo', 'wells'}")

   # Training settings
   train = parser.add_argument_group('train', description = "training settings")
   train.add_argument('--num_steps', type=int, default=20)
   train.add_argument('--shots', type=int, default=1024)
   train.add_argument('--db', type=float, default=0.1)
   train.add_argument('--delta', type=float, default=0.1)

   # Debugging settings
   debug = parser.add_argument_group('debug', description = "debug settings")
   debug.add_argument('--fpath', type=str, default=None)
   debug.add_argument('--log_interval', type=int, default=10)
   debug.add_argument('--plot_type', type=str, default='none', help="{'none', 'energy'}")

   args = parser.parse_args()
   return args

demo_args = SimpleNamespace(
   model_name = 'QITE',
   num_qubits = 1,
   task = 'demo',
   num_steps = 20,
   shots = 1024,
   db = 0.1,
   delta = 0.1,
   fpath = None,
   log_interval = 10,
   plot_type = 'none'
)