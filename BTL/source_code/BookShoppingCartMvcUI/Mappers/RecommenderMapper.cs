using BookShoppingCartMvcUI.Models.Recommendation;
using BookShoppingCartMvcUI.Models.ViewModels.Recommendation;

namespace BookShoppingCartMvcUI.Mappers
{
    public static class RecommenderMapper
    {
        // ==================================================
        // 1️⃣ CONFIG
        // ==================================================
        public static ConfigVM ToVM(this RecommenderConfigDto dto)
        {
            if (dto == null)
                return new ConfigVM
                {
                    DataStatistics = new DataStatisticsVM(),
                    UserThresholds = new UserThresholdVM(),
                    BehaviorWeights = new BehaviorWeightVM
                    {
                        Rating = new Dictionary<int, int>()
                    }
                };

            return new ConfigVM
            {
                DataStatistics = new DataStatisticsVM
                {
                    TotalInteractions = dto.Data_Statistics?.Total_Interactions ?? 0,
                    NumUsers = dto.Data_Statistics?.Num_Users ?? 0,
                    NumItems = dto.Data_Statistics?.Num_Items ?? 0,
                    NumRatings = dto.Data_Statistics?.Num_Ratings ?? 0,
                    NumScores = dto.Data_Statistics?.Num_Scores ?? 0
                },

                UserThresholds = new UserThresholdVM
                {
                    NLowMax = dto.User_Thresholds?.N_Low_Max ?? 0,
                    NMediumMax = dto.User_Thresholds?.N_Medium_Max ?? 0
                },

                BehaviorWeights = new BehaviorWeightVM
                {
                    View = dto.Behavior_Weights?.View ?? 0,
                    AddToCart = dto.Behavior_Weights?.AddToCart ?? 0,
                    Purchase = dto.Behavior_Weights?.Purchase ?? 0,
                    Rating = dto.Behavior_Weights?.Rating?
                        .ToDictionary(
                            x => int.Parse(x.Key),
                            x => x.Value
                        ) ?? new Dictionary<int, int>()
                }
            };
        }

        // ==================================================
        // 2️⃣ PERSONAL RECOMMENDATION
        // ==================================================
        public static PersonalRecommendVM ToVM(this RecommendationResult dto)
        {
            if (dto == null)
                return new PersonalRecommendVM
                {
                    Recommendations = new List<RecommendationItemVM>()
                };

            return new PersonalRecommendVM
            {
                Strategy = dto.Strategy ?? "",
                Reason = dto.Reason ?? "",

                Profile = dto.Profile == null
                    ? new UserProfileVM()
                    : new UserProfileVM
                    {
                        TotalInteractions = dto.Profile.N_Interaction,
                        UserType = dto.Profile.Type
                    },

                Recommendations = dto.Recommendations?
                    .Select(x => new RecommendationItemVM
                    {
                        BookId = x.BookId,
                        Score = x.Score
                    })
                    .ToList()
                    ?? new List<RecommendationItemVM>()
            };
        }


        // ==================================================
        // 3️⃣ MODEL EVALUATION
        // ==================================================
        public static EvaluationVM ToVM(this EvaluationMetricsDto dto)
        {
            var vm = new EvaluationVM
            {
                Metrics = new Dictionary<string, MetricVM>()
            };

            if (dto?.Metrics == null)
                return vm;

            foreach (var kv in dto.Metrics)
            {
                var metric = kv.Value; // MetricScoreDto
                if (metric == null) continue;

                vm.Metrics[kv.Key] = new MetricVM
                {
                    PrecisionAtK = metric.Precision_At_K,
                    RecallAtK = metric.Recall_At_K,
                    NdcgAtK = metric.NDCG_At_K
                };
            }

            return vm;
        }
    }
}
