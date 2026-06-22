#!/usr/bin/env python3
"""E7 stub → delegates to exp_evaluation.py"""
import sys, os
sys.argv = [sys.argv[0], "--exp", "e7"] + sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "exp_evaluation.py")).read())
