from subprocess import run

cmd1 = "gcc -o mnist_nn main.c nn.c mnist_loader.c -lm -Wall -Wextra"
cmd2 = "./mnist_nn"

run(cmd1, shell=True)
run(cmd2, shell=True)