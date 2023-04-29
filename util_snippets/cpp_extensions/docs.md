tutorials from https://pytorch.org/tutorials/advanced/cpp_extension.html  
  
to build, from the lltm-extensions directory, run: `python setup.py install`  
then run the pythno vs cpp numbers using:  
`python main.py --py --num_iter 10000 --batch_size 16 --input_size 32 --state_size 128`  
`python main.py --cpp --num_iter 10000 --batch_size 16 --input_size 32 --state_size 128`  

