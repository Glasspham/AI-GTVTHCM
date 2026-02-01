using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class EvaluationMetricsDto
    {
        public string Best_Model { get; set; }

        public Dictionary<string, MetricScoreDto> Metrics { get; set; }
    }

    public class MetricScoreDto
    {
        [JsonPropertyName("Precision@K")]
        public double Precision_At_K { get; set; }

        [JsonPropertyName("Recall@K")]
        public double Recall_At_K { get; set; }

        [JsonPropertyName("NDCG@K")]
        public double NDCG_At_K { get; set; }
    }
}
