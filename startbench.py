import warnings,time
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import shutil
import gpt_2_simple as gpt2
try:
	shutil.rmtree('./checkpoint')
except:
	pass

gpt2.download_gpt2(model_name="117M")
sess = gpt2.start_tf_sess()
t1=time.time()
print("GPT 117M BENCHMARK STARTED!")
gpt2.finetune(sess,
              'shakespeare.txt',
              model_name="117M",
              steps=1000) 
gpt2.generate(sess)
print("GPT 117M BENCHMARK ENDED!")  
t2=time.time()-t1
print("BENCHMARK TIME: " + str(t2) + " seconds")
gpt117t=t2

try:
	shutil.rmtree('./checkpoint')
except:
	pass

gpt2.download_gpt2(model_name="345M")
sess = gpt2.start_tf_sess()
t1=time.time()
print("GPT 345M BENCHMARK STARTED!")
gpt2.finetune(sess,
              'shakespeare.txt',
              model_name="345M",
              steps=1000) 
gpt2.generate(sess)
print("GPT 345M BENCHMARK ENDED!")  
t2=time.time()-t1
gpt345t=t2

print("BENCHMARK 117M TIME: " + str(gpt117t) + " seconds")
print("BENCHMARK 345M TIME: " + str(gpt345t) + " seconds")