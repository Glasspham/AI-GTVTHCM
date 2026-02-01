using Microsoft.AspNetCore.Mvc;

namespace BookShoppingCartMvcUI.Controllers
{
    public class RecommendSystemController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}
