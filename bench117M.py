import warnings,time
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import shutil
import gpt_2_simple as gpt2
try:
	shutil.rmtree('./checkpoint')
except:
	pass

sess = gpt2.start_tf_sess()
t1=time.time()
print("GPT BENCHMARK STARTED!")
gpt2.finetune(sess,
              'shakespeare.txt',
              model_name="117M",
              steps=1000) 
gpt2.generate(sess)
print("GPT BENCHMARK ENDED!")  
t2=time.time()-t1
print("BENCHMARK TIME: " + str(t2) + " seconds")