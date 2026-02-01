using BookShoppingCartMvcUI.Mappers;
using BookShoppingCartMvcUI.Models.ViewModels;
using BookShoppingCartMvcUI.Models.ViewModels.Recommendation;
using BookShoppingCartMvcUI.Services;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;

namespace BookShoppingCartMvcUI.Controllers
{
    public class RecommendDashboardController : Controller
    {
        private readonly RecommendationService _recommendationService;
        private readonly UserManager<IdentityUser> _userManager;

        public RecommendDashboardController(
            RecommendationService recommendationService,
            UserManager<IdentityUser> userManager)
        {
            _recommendationService = recommendationService;
            _userManager = userManager;
        }

        private string? GetUserId()
            => User.Identity != null && User.Identity.IsAuthenticated
                ? _userManager.GetUserId(User)
                : null;

        // ======================================================
        // ===================== INDEX ==========================
        // ======================================================
        public async Task<IActionResult> Index()
        {
            var currentUserId = GetUserId();
            if (currentUserId == null)
                return Unauthorized();

            // 1️⃣ Config
            var configDto =
                await _recommendationService.GetRecommenderConfigAsync(currentUserId);

            // 2️⃣ Evaluation
            var evalDto =
                await _recommendationService.GetEvaluationMetricsAsync();

            // 3️⃣ User list (🔥 CÁI BẠN THIẾU)
            var users =
                await _recommendationService.GetUsersAsync();

            var vm = new RecommendDashboardVM
            {
                Config = configDto.ToVM(),
                Evaluation = evalDto.ToVM(),

                PersonalPage = new PersonalRecommendPageVM
                {
                    Users = users.Select(u => new UserSelectVM
                    {
                        UserId = u.UserId,
                        Label = $"{u.UserId[..8]}... ({u.TotalInteractions} interactions)"
                    }).ToList()
                }
            };

            return View(vm);
        }

        // ======================================================
        // ============ PERSONAL RESULT (AJAX) ==================
        // ======================================================
        [HttpGet]
        public async Task<IActionResult> GetPersonalResult(string userId)
        {
            if (string.IsNullOrWhiteSpace(userId))
                return PartialView("_PersonalResult", null);

            var dto =
                await _recommendationService.GetPersonalAsync(userId);

            var vm = dto.ToVM();

            return PartialView("_PersonalResult", vm);
        }
    }
}
