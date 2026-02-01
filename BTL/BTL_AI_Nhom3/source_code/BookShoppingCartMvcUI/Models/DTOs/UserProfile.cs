namespace BookShoppingCartMvcUI.Services
{
    public class UserProfile
    {
        public string Type { get; set; }
        public int N_Interaction { get; set; }
        public double Total_Score { get; set; }
        public double Avg_Score { get; set; }
        public bool Has_Rating { get; set; }
        public int N_Positive { get; set; }
        public int N_Negative { get; set; }
    }

    public class AlgorithmParameters
    {
        public string Model { get; set; }
        public int Top_N { get; set; }
    }

    public class UserStatistics
    {
        public bool User_Exists { get; set; }
        public int Num_User_Ratings { get; set; }
        public int Num_User_Scores { get; set; }
        public string[] Rated_Items { get; set; }
        public string[] Implicit_Items { get; set; }
    }
}
