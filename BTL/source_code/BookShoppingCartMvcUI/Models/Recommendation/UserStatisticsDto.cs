using System.Collections.Generic;

namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class UserStatisticsDto
    {
        public bool User_Exists { get; set; }
        public int Num_User_Ratings { get; set; }
        public int Num_User_Scores { get; set; }

        public List<string> Rated_Items { get; set; } = new();
        public List<string> Implicit_Items { get; set; } = new();
    }
}
