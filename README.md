# poke-trainer


```sh
$ git clone https://github.com/kywch/poke-trainer.git
$ cd poke-trainer
$ python -m venv --system-site-packages .venv
$ source .venv/bin/activate
$ pip install -e .
```

## Troubleshooting

If build/compile fails or you get open-cv related error
```
$ sudo apt-get update
$ sudo apt-get install build-essential libgl1
```

If making venv fails, check the python version (e.g, 3.11)
```
$ sudo apt-get install python3.11-venv
```