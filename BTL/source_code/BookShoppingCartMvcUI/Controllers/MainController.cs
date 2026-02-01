using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BookShoppingCartMvcUI.Models;
using BookShoppingCartMvcUI.Models.DTOs;
using BookShoppingCartMvcUI.Models.Recommendation;
using BookShoppingCartMvcUI.Services;
using Microsoft.AspNetCore.Identity;

namespace BookShoppingCartMvcUI.Controllers
{
    public class MainController : Controller
    {
        private readonly ApplicationDbContext _context;
        private readonly RecommendationService _recommendationService;
        private readonly IBookRepository _bookRepository;
        private readonly UserManager<IdentityUser> _userManager;

        public MainController(
            ApplicationDbContext context,
            RecommendationService recommendationService,
            IBookRepository bookRepository,
            UserManager<IdentityUser> userManager)
        {
            _context = context;
            _recommendationService = recommendationService;
            _bookRepository = bookRepository;
            _userManager = userManager;
        }

        // ================== USER ID ==================
        private string? GetCurrentUserId()
            => User.Identity != null && User.Identity.IsAuthenticated
                ? _userManager.GetUserId(User)
                : null;

        // ================== MAP BOOK ==================
        private async Task<List<BookDTO>> MapRecommendationsAsync(
            IEnumerable<RecommendItem> recommendations)
        {
            var result = new List<BookDTO>();

            foreach (var rec in recommendations)
            {
                var book = await _bookRepository.GetBookById(rec.BookId);
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

        // ================== INDEX ==================
        public async Task<IActionResult> Index()
        {
            // ===== 1. BOOKS BY GENRE (CONTENT-BASED) =====
            var genres = await _context.Genres.ToListAsync();
            var sections = new List<BooksByGenreSectionModel>();

            foreach (var genre in genres)
            {
                var books = await _context.Books
                    .Where(b => b.GenreId == genre.Id)
                    .Take(10)
                    .Select(b => new BookDTO
                    {
                        Id = b.Id,
                        BookName = b.BookName,
                        AuthorName = b.AuthorName,
                        Price = b.Price,
                        Image = b.Image,
                        GenreId = b.GenreId,
                        GenreName = genre.GenreName
                    })
                    .ToListAsync();

                if (books.Any())
                {
                    sections.Add(new BooksByGenreSectionModel
                    {
                        GenreId = genre.Id,
                        GenreName = genre.GenreName,
                        Books = books
                    });
                }
            }

            // ===== 2. PERSONALIZED RECOMMENDATION =====
            try
            {
                var userId = GetCurrentUserId(); // null nếu chưa login

                var recResult = await _recommendationService
                    .GetPersonalAsync(userId);

                var recommendedBooks = await MapRecommendationsAsync(
                    recResult.Recommendations
                        .DistinctBy(r => r.BookId)
                );

                ViewBag.RecommendedBooks = recommendedBooks;
                ViewBag.RecommendStrategy = recResult.Strategy;
                ViewBag.RecommendReason = recResult.Reason;
                ViewBag.RecommendProfile = recResult.Profile;
            }
            catch (Exception ex)
            {
                // ❌ Recommender chết → web vẫn sống
                Console.WriteLine("❌ RECOMMEND ERROR: " + ex.Message);

                ViewBag.RecommendedBooks = new List<BookDTO>();
                ViewBag.RecommendStrategy = "NONE";
                ViewBag.RecommendReason = "Recommender service unavailable";
            }

            return View(sections);
        }
    }
}
