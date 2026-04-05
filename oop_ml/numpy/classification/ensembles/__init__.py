"""Predicting a class from many models rather than one.

Members vote by averaging their probabilities rather than their labels. A hard
vote throws away how sure each member was, so a member that barely favoured a
class would count as much as one that was overwhelmingly certain.
"""
