namespace BookShoppingCartMvcUI.Models.ViewModels.Recommendation
{
    public class ConfigVM
    {
        public DataStatisticsVM DataStatistics { get; set; }
        public UserThresholdVM UserThresholds { get; set; }
        public BehaviorWeightVM BehaviorWeights { get; set; }
    }

    public class DataStatisticsVM
    {
        public int TotalInteractions { get; set; }
        public int NumUsers { get; set; }
        public int NumItems { get; set; }
        public int NumRatings { get; set; }
        public int NumScores { get; set; }
    }

    public class UserThresholdVM
    {
        public int NLowMax { get; set; }
        public int NMediumMax { get; set; }
    }

    public class BehaviorWeightVM
    {
        public int View { get; set; }
        public int AddToCart { get; set; }
        public int Purchase { get; set; }
        public Dictionary<int, int> Rating { get; set; }
    }
}
