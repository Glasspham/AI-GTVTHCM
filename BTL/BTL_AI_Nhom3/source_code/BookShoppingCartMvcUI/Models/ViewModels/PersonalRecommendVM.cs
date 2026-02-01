namespace BookShoppingCartMvcUI.Models.ViewModels.Recommendation;
public class PersonalRecommendVM
{
    public string Strategy { get; set; } = "";
    public string Reason { get; set; } = "";

    public UserProfileVM Profile { get; set; } = new();

    public List<RecommendationItemVM> Recommendations { get; set; } = new();
}

public class UserProfileVM
{
    public int TotalInteractions { get; set; }
    public string UserType { get; set; } // cold / warm / heavy
}

public class RecommendationItemVM
{
    public int BookId { get; set; }
    public string BookName { get; set; }
    public double Score { get; set; }
}
