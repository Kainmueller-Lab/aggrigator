import inspect

from enum import Enum

from aggrigator.methods import AggregationMethods

### CUSTOM STRATEGY LISTS ###
# NOTE: Define your custom strategy lists here.
class StrategyList(Enum):
    ALL_DEFAULT = [(name, None) for name, method in inspect.getmembers(AggregationMethods, predicate=inspect.isfunction)]
    BASIC = [(AggregationMethods.mean, None),
             (AggregationMethods.sum, None),
             (AggregationMethods.max, None)]
    SPATIAL = [(AggregationMethods.morans_I, None),
               (AggregationMethods.gearys_C, None)]
    CLASS_MEAN = [(AggregationMethods.class_mean, 0),
                  (AggregationMethods.class_mean, 1)]
    THRESHOLD = [(AggregationMethods.above_threshold_mean, 0.2),
                 (AggregationMethods.above_threshold_mean, 0.4),
                 (AggregationMethods.above_threshold_mean, 0.6),
                 (AggregationMethods.above_threshold_mean, 0.8),
                 (AggregationMethods.above_threshold_mean, 0.9),
                 ]
    
    QUANTILE = [(AggregationMethods.above_quantile_mean, 0.2),
                (AggregationMethods.above_quantile_mean, 0.4),
                (AggregationMethods.above_quantile_mean, 0.6),
                (AggregationMethods.above_quantile_mean, 0.8),
                (AggregationMethods.above_quantile_mean, 0.9),]
