namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class UserProfileDto
    {
        public string Type { get; set; }          // RICH_DATA, MEDIUM_DATA, COLD_START
        public int N_Interaction { get; set; }
        public double Total_Score { get; set; }
        public double Avg_Score { get; set; }
        public bool Has_Rating { get; set; }
        public int N_Positive { get; set; }
        public int N_Negative { get; set; }
    }
}
