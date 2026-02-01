using System;
using System.Buffers.Text;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using BookShoppingCartMvcUI.Models.Recommendation;

namespace BookShoppingCartMvcUI.Services
{
    public class RecommendationService
    {
        private readonly HttpClient _httpClient;
        private const string BASE_URL = "http://127.0.0.1:8000";

        public RecommendationService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        // =========================================================
        // =============== CORE HTTP HELPER ========================
        // =========================================================
        private async Task<T> GetAsync<T>(string url)
        {
            Console.WriteLine("===== CALL API =====");
            Console.WriteLine(url);

            var response = await _httpClient.GetAsync(url);

            if (!response.IsSuccessStatusCode)
            {
                var err = await response.Content.ReadAsStringAsync();
                throw new Exception($"API Error: {response.StatusCode} - {err}");
            }

            var json = await response.Content.ReadAsStringAsync();

            return JsonSerializer.Deserialize<T>(
                json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
            )!;
        }

        // =========================================================
        // =============== CÁC MODEL CŨ (GIỮ NGUYÊN) ===============
        // =========================================================

        public Task<RecommendationResult> GetUserCFAsync(string userId)
            => GetAsync<RecommendationResult>(
                $"{BASE_URL}/recommend/usercf/{userId}"
            );

        public Task<RecommendationResult> GetItemCFAsync(string userId)
            => GetAsync<RecommendationResult>(
                $"{BASE_URL}/recommend/itemcf/{userId}"
            );

        public Task<RecommendationResult> GetMFAsync(string userId)
            => GetAsync<RecommendationResult>(
                $"{BASE_URL}/recommend/mf/{userId}"
            );

        public Task<RecommendationResult> GetBestModelAsync(string userId)
            => GetAsync<RecommendationResult>(
                $"{BASE_URL}/recommend/best/{userId}"
            );

        // =========================================================
        // =============== PERSONAL / ANONYMOUS ====================
        // =========================================================

        public async Task<RecommendationResult> GetPersonalAsync(string? userId)
        {
            // ===== USER VÔ DANH =====
            if (string.IsNullOrEmpty(userId))
            {
                return await GetAsync<RecommendationResult>(
                    $"{BASE_URL}/recommend/anonymous"
                );
            }

            // ===== USER ĐÃ LOGIN =====
            return await GetAsync<RecommendationResult>(
                $"{BASE_URL}/recommend/personal/{userId}"
            );
        }

        // =========================================================
        // =============== DASHBOARD / ADMIN =======================
        // =========================================================

        /// <summary>
        /// Lấy config + statistics dùng cho dashboard
        /// </summary>
        /// // danh sách user
        public Task<List<UserSummaryDto>> GetUsersAsync()
            => GetAsync<List<UserSummaryDto>>(
                $"{BASE_URL}/users"
            );

        public Task<RecommenderConfigDto> GetRecommenderConfigAsync(string userId)
            => GetAsync<RecommenderConfigDto>(
                $"{BASE_URL}/recommender/config/{userId}"
            );

        /// <summary>
        /// Lấy các chỉ số đánh giá thuật toán
        /// </summary>
        public Task<EvaluationMetricsDto> GetEvaluationMetricsAsync()
            => GetAsync<EvaluationMetricsDto>(
                $"{BASE_URL}/evaluate"
            );
    }
}
