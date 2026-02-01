using BookShoppingCartMvcUI.Models.DTOs;
using BookShoppingCartMvcUI.Models.Recommendation;
using BookShoppingCartMvcUI.Services;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;

namespace BookShoppingCartMvcUI.Controllers
{
    public class RecommendationController : Controller
    {
        private readonly RecommendationService _recommendationService;
        private readonly IBookRepository _bookRepository;
        private readonly UserManager<IdentityUser> _userManager;

        public RecommendationController(
            RecommendationService recommendationService,
            IBookRepository bookRepository,
            UserManager<IdentityUser> userManager)
        {
            _recommendationService = recommendationService;
            _bookRepository = bookRepository;
            _userManager = userManager;
        }

        // ======================================================
        // =================== HELPERS ==========================
        // ======================================================

        private string? GetCurrentUserId()
            => _userManager.GetUserId(User);

        private async Task<List<BookDTO>> MapToBooksAsync(
            IEnumerable<RecommendItem> recommendations)
        {
            var books = new List<BookDTO>();

            foreach (var rec in recommendations)
            {
                var book = await _bookRepository.GetBookById(rec.BookId);
                if (book == null) continue;

                books.Add(new BookDTO
                {
                    Id = book.Id,
                    BookName = book.BookName,
                    AuthorName = book.AuthorName,
                    Price = book.Price,
                    Image = book.Image,
                    GenreId = book.GenreId,
                    GenreName = book.Genre?.GenreName,
                    Description = book.Description
                });
            }

            return books;
        }

        // ======================================================
        // =================== CORE ACTION ======================
        // ======================================================

        /// <summary>
        /// Endpoint duy nhất cho FE gọi gợi ý
        /// - Login  → personal recommend
        /// - Logout → popular books
        /// </summary>
        [HttpGet]
        public async Task<IActionResult> Personal()
        {
            var userId = GetCurrentUserId();

            RecommendationResult result;
            try
            {
                result = await _recommendationService.GetPersonalAsync(userId);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Recommend error: {ex.Message}");
            }

            var books = await MapToBooksAsync(result.Recommendations);

            // 👉 Nếu dùng Ajax
            return Json(new
            {
                strategy = result.Strategy,
                reason = result.Reason,
                profile = result.Profile,
                books
            });

            // 👉 Nếu render Razor View thì dùng:
            // return View(books);
        }

        // ======================================================
        // ========== OPTIONAL: ENDPOINT RIÊNG (GIỮ LẠI) =========
        // ======================================================

        [HttpGet]
        public async Task<IActionResult> UserCF()
        {
            var userId = GetCurrentUserId();
            if (userId == null) return Unauthorized();

            var result = await _recommendationService.GetUserCFAsync(userId);
            var books = await MapToBooksAsync(result.Recommendations);

            return Json(books);
        }

        [HttpGet]
        public async Task<IActionResult> ItemCF()
        {
            var userId = GetCurrentUserId();
            if (userId == null) return Unauthorized();

            var result = await _recommendationService.GetItemCFAsync(userId);
            var books = await MapToBooksAsync(result.Recommendations);

            return Json(books);
        }

        [HttpGet]
        public async Task<IActionResult> MF()
        {
            var userId = GetCurrentUserId();
            if (userId == null) return Unauthorized();

            var result = await _recommendationService.GetMFAsync(userId);
            var books = await MapToBooksAsync(result.Recommendations);

            return Json(books);
        }
    }
}
