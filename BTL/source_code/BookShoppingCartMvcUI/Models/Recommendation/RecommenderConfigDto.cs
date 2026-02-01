namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class RecommenderConfigDto
    {
        public DataStatisticsDto Data_Statistics { get; set; }
        public UserThresholdsDto User_Thresholds { get; set; }
        public BehaviorWeightsDto Behavior_Weights { get; set; }
    }
    public class DataStatisticsDto
    {
        public int Total_Interactions { get; set; }
        public int Num_Users { get; set; }
        public int Num_Items { get; set; }
        public int Num_Ratings { get; set; }
        public int Num_Scores { get; set; }
    }
    public class UserThresholdsDto
    {
        public int N_Low_Max { get; set; }
        public int N_Medium_Max { get; set; }
    }
    public class BehaviorWeightsDto
    {
        public int View { get; set; }
        public int AddToCart { get; set; }
        public int Purchase { get; set; }

        public Dictionary<string, int> Rating { get; set; }
    }
}
