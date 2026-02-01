using BookShoppingCartMvcUI.Data;
using BookShoppingCartMvcUI.Models;
using BookShoppingCartMvcUI.Models.DTOs;
using BookShoppingCartMvcUI.Models.Recommendation;
using BookShoppingCartMvcUI.Services;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace BookShoppingCartMvcUI.Controllers
{
    public class DetailedBookController : Controller
    {
        private readonly IBookRepository _bookRepo;
        private readonly ApplicationDbContext _context;
        private readonly UserManager<IdentityUser> _userManager;
        private readonly RecommendationService _recommendationService;

        public DetailedBookController(
            IBookRepository bookRepo,
            ApplicationDbContext context,
            UserManager<IdentityUser> userManager,
            RecommendationService recommendationService)
        {
            _bookRepo = bookRepo;
            _context = context;
            _userManager = userManager;
            _recommendationService = recommendationService;
        }

        public async Task<IActionResult> Details(int id)
        {
            var book = await _bookRepo.GetBookById(id);
            if (book == null)
                return NotFound();

            // ==================================================
            // ========== GHI NHẬN INTERACTION ==================
            // ==================================================
            if (User.Identity!.IsAuthenticated)
            {
                var userId = _userManager.GetUserId(User);

                var existing = await _context.UserInteractions
                    .FirstOrDefaultAsync(x => x.UserId == userId && x.BookId == id);

                if (existing == null)
                {
                    _context.UserInteractions.Add(new UserInteraction
                    {
                        UserId = userId,
                        BookId = id,
                        Score = 1,
                        InteractionDate = DateTime.Now
                    });
                }
                else
                {
                    existing.Score = (existing.Score ?? 0) + 1;
                    existing.InteractionDate = DateTime.Now;
                    _context.UserInteractions.Update(existing);
                }

                await _context.SaveChangesAsync();

                // ==================================================
                // ========== USER CF RECOMMEND =====================
                // ==================================================
                var userCfResult = await _recommendationService
                    .GetUserCFAsync(userId);

                ViewBag.UserCFRecommendations =
                    await MapRecommendationsAsync(userCfResult.Recommendations);

                // ==================================================
                // ========== ITEM CF RECOMMEND =====================
                // ==================================================
                var itemCfResult = await _recommendationService
                    .GetItemCFAsync(userId);

                ViewBag.ItemCFRecommendations =
                    await MapRecommendationsAsync(itemCfResult.Recommendations);
            }

            // ==================================================
            // ========== RELATED BOOKS (CONTENT-BASED) ==========
            // ==================================================
            var relatedBooks = await _bookRepo.GetBooksByGenreId(book.GenreId);

            ViewBag.RelatedBooks = new BooksByGenreSectionModel
            {
                GenreId = book.GenreId,
                GenreName = book.Genre?.GenreName ?? "Sản phẩm liên quan",
                Books = relatedBooks
                    .Where(b => b.Id != book.Id)
                    .Select(b => new BookDTO
                    {
                        Id = b.Id,
                        BookName = b.BookName,
                        AuthorName = b.AuthorName,
                        Price = b.Price,
                        Image = b.Image,
                        GenreId = b.GenreId,
                        GenreName = b.Genre?.GenreName,
                        Description = b.Description
                    }).ToList()
            };

            return View(book);
        }

        // ==================================================
        // ========== HELPER MAP =============================
        // ==================================================
        private async Task<List<BookDTO>> MapRecommendationsAsync(
            IEnumerable<RecommendItem> recommendations)
        {
            var result = new List<BookDTO>();

            foreach (var rec in recommendations)
            {
                var book = await _bookRepo.GetBookById(rec.BookId);
                if (book == null) continue;

                result.Add(new BookDTO
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

            return result;
        }
    }
}
