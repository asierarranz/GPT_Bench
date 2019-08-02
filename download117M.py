import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gpt_2_simple as gpt2
gpt2.download_gpt2(model_name="117M")
