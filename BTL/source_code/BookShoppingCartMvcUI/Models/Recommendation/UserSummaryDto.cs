namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class UserSummaryDto
    {
        public string UserId { get; set; }
        public int TotalInteractions { get; set; }
        public int NumRatings { get; set; }
        public int NumScores { get; set; }
    }
}
