from sagent.metrics.base import BaseMetric
from sagent.metrics.max import Max
from sagent.metrics.mean import Mean
from sagent.metrics.min import Min
from sagent.metrics.sum import Sum
from sagent.metrics.uv_script import UvScript
from sagent.models.metric.type import MetricType


class MetricFactory:
    _METRICS: list[type[BaseMetric]] = [
        Sum,
        Min,
        Max,
        Mean,
        UvScript,
    ]
    _METRIC_MAP: dict[MetricType, type[BaseMetric]] = {
        MetricType.SUM: Sum,
        MetricType.MIN: Min,
        MetricType.MAX: Max,
        MetricType.MEAN: Mean,
        MetricType.UV_SCRIPT: UvScript,
    }

    @classmethod
    def create_metric(
        cls,
        metric_type: MetricType,
        **kwargs,
    ) -> BaseMetric:
        """
        Create a metric from a metric type.

        Args:
            metric_type (MetricType): The type of the metric.
            **kwargs: Additional keyword arguments to pass to the metric constructor.

        Returns:
            BaseMetric: The created metric.

        Raises:
            ValueError: If the metric type is invalid or required parameters are missing.
        """
        if metric_type not in cls._METRIC_MAP:
            raise ValueError(
                f"Unsupported metric type: {metric_type}. This could be because the "
                "metric is not registered in the MetricFactory or because the metric "
                "type is invalid."
            )

        return cls._METRIC_MAP[metric_type](**kwargs)
