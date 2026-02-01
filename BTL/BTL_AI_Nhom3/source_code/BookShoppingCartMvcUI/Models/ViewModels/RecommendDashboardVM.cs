namespace BookShoppingCartMvcUI.Models.ViewModels.Recommendation
{
    public class RecommendDashboardVM
    {
        public ConfigVM Config { get; set; } = new();
        public EvaluationVM Evaluation { get; set; } = new();

        // 🔥 Gộp personal vào index
        public PersonalRecommendPageVM PersonalPage { get; set; } = new();
    }
}
