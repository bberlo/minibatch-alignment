#!/bin/bash

rm -rf results/*
rm -rf Streams/*
rm -rf tmp/*

python3 experiment_automation.py -m_n std -t tuning -g 0 -d widar -d_t dfs -bb efficientnet
python3 experiment_automation.py -m_n std -t tuning -g 0 -d widar -d_t gaf -bb efficientnet
python3 experiment_automation.py -m_n wigrunt -t tuning -g 0 -d widar -d_t dfs -bb efficientnet
python3 experiment_automation.py -m_n wigrunt -t tuning -g 0 -d widar -d_t gaf -bb efficientnet

python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t position
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t position
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t environment
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t environment

python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t position
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t position
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t environment
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t environment

python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t position
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t position
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t environment
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t environment

python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t position
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t position
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t environment
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t environment

python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t position
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t position
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t user
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d signfi -d_t dfs -bb efficientnet -do_t environment
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t user
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d signfi -d_t gaf -bb efficientnet -do_t environment


python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation -lft 2
python3 experiment_automation.py -m_n std -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation -lft 2

python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation -lft 2
python3 experiment_automation.py -m_n domain_class -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation -lft 2

python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation -lft 2
python3 experiment_automation.py -m_n wigrunt -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation -lft 2

python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation -lft 2
python3 experiment_automation.py -m_n minibatch -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation -lft 2

python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t dfs -bb efficientnet -do_t orientation -lft 2
python3 experiment_automation.py -m_n fido -t testing-leave-out -g 0 -d widar -d_t gaf -bb efficientnet -do_t orientation -lft 2
