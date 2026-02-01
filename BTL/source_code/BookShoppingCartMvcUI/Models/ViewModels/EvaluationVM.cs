public class EvaluationVM
{
    public Dictionary<string, MetricVM> Metrics { get; set; }
}

public class MetricVM
{
    public double PrecisionAtK { get; set; }
    public double RecallAtK { get; set; }
    public double NdcgAtK { get; set; }
}
