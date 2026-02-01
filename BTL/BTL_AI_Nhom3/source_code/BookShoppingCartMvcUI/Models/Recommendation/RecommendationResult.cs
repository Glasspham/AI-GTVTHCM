using System.Collections.Generic;

namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class RecommendationResult
    {
        public string Strategy { get; set; }
        public string Reason { get; set; }

        public UserProfileDto Profile { get; set; }
        public AlgorithmParametersDto Algorithm_Parameters { get; set; }
        public UserStatisticsDto User_Statistics { get; set; }

        public List<RecommendItem> Recommendations { get; set; } = new();
    }
}
