using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;

namespace BookShoppingCartMvcUI.Data;

public class DbSeeder
{
    public static async Task SeedDefaultData(IServiceProvider service)
    {
        try
        {
            var context = service.GetService<ApplicationDbContext>();

            // this block will check if there are any pending migrations and apply them
            if ((await context.Database.GetPendingMigrationsAsync()).Count() > 0)
            {
                await context.Database.MigrateAsync();
            }

            var userMgr = service.GetService<UserManager<IdentityUser>>();
            var roleMgr = service.GetService<RoleManager<IdentityRole>>();

            // create admin role if not exists
            var adminRoleExists = await roleMgr.RoleExistsAsync(Roles.Admin.ToString());

            if (!adminRoleExists)
            {
                await roleMgr.CreateAsync(new IdentityRole(Roles.Admin.ToString()));
            }

            // create user role if not exists
            var userRoleExists = await roleMgr.RoleExistsAsync(Roles.User.ToString());

            if (!userRoleExists)
            {
                await roleMgr.CreateAsync(new IdentityRole(Roles.User.ToString()));
            }

            // create admin user

            var admin = new IdentityUser
            {
                UserName = "admin@gmail.com",
                Email = "admin@gmail.com",
                EmailConfirmed = true
            };

            var userInDb = await userMgr.FindByEmailAsync(admin.Email);
            if (userInDb is null)
            {
                await userMgr.CreateAsync(admin, "Admin@123");
                await userMgr.AddToRoleAsync(admin, Roles.Admin.ToString());
            }


            if (!context.Genres.Any())
            {
                await SeedGenreAsync(context);
            }

            if (!context.Books.Any())
            {
                await SeedBooksAsync(context);
                // update stock table
                await context.Database.ExecuteSqlRawAsync(@"
                     INSERT INTO Stock(BookId,Quantity) 
                     SELECT 
                     b.Id,
                     10 
                     FROM Book b
                     WHERE NOT EXISTS (
                     SELECT * FROM [Stock]
                     );
           ");
            }
            else
            {
                // 🛠️ Cập nhật description cho các sách cũ nếu bị NULL
                await UpdateMissingDescriptionsAsync(context);
            }

            if (!context.orderStatuses.Any())
            {
                await SeedOrderStatusAsync(context);
            }

            // Create stored procedures if они missing
            await SeedStoredProceduresAsync(context);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
        }
    }

    private static async Task SeedStoredProceduresAsync(ApplicationDbContext context)
    {
        try
        {
            // 🛠️ 1. Fix UserInteractions Schema
            await context.Database.ExecuteSqlRawAsync(@"
                IF EXISTS (SELECT * FROM sys.tables WHERE name = 'UserInteractions')
                BEGIN
                    IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('UserInteractions') AND name = 'Score')
                    BEGIN
                        ALTER TABLE UserInteractions ADD Score FLOAT NULL;
                    END
                    IF EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('UserInteractions') AND name = 'InteractionType')
                    BEGIN
                        ALTER TABLE UserInteractions ALTER COLUMN InteractionType NVARCHAR(MAX) NULL;
                    END
                END
            ");

            // 🛠️ 2. Create Python API Metadata Tables
            await context.Database.ExecuteSqlRawAsync(@"
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'RecommenderModel')
                BEGIN
                    CREATE TABLE RecommenderModel (
                        model_id INT PRIMARY KEY,
                        model_name NVARCHAR(50) NOT NULL,
                        top_n INT DEFAULT 5,
                        is_active BIT DEFAULT 1
                    );
                    INSERT INTO RecommenderModel (model_id, model_name, top_n, is_active) VALUES 
                    (1, 'user_cf', 5, 1),
                    (2, 'item_cf', 5, 1),
                    (3, 'mf', 5, 1);
                END

                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'CF_Model_Params')
                BEGIN
                    CREATE TABLE CF_Model_Params (
                        model_id INT PRIMARY KEY,
                        k INT DEFAULT 10,
                        alpha FLOAT DEFAULT 0.5,
                        FOREIGN KEY (model_id) REFERENCES RecommenderModel(model_id)
                    );
                    INSERT INTO CF_Model_Params (model_id, k, alpha) VALUES 
                    (1, 10, 0.5),
                    (2, 10, 0.5);
                END

                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'MF_Model_Params')
                BEGIN
                    CREATE TABLE MF_Model_Params (
                        model_id INT PRIMARY KEY,
                        latent_k INT DEFAULT 20,
                        learning_rate FLOAT DEFAULT 0.01,
                        reg_lambda FLOAT DEFAULT 0.1,
                        n_iter INT DEFAULT 50,
                        weight_rating FLOAT DEFAULT 1.0,
                        weight_score FLOAT DEFAULT 0.5,
                        pred_min FLOAT DEFAULT 1.0,
                        pred_max FLOAT DEFAULT 5.0,
                        FOREIGN KEY (model_id) REFERENCES RecommenderModel(model_id)
                    );
                    INSERT INTO MF_Model_Params (model_id) VALUES (3);
                END

                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'UserBehaviorWeights')
                BEGIN
                    CREATE TABLE UserBehaviorWeights (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        n_low_max INT DEFAULT 5,
                        n_medium_max INT DEFAULT 10,
                        weight_view INT DEFAULT 1,
                        weight_addtocart INT DEFAULT 3,
                        weight_purchase INT DEFAULT 5,
                        weight_rating_1 INT DEFAULT 1,
                        weight_rating_2 INT DEFAULT 2,
                        weight_rating_3 INT DEFAULT 3,
                        weight_rating_4 INT DEFAULT 4,
                        weight_rating_5 INT DEFAULT 5,
                        created_at DATETIME DEFAULT GETDATE()
                    );
                    INSERT INTO UserBehaviorWeights (n_low_max, n_medium_max) VALUES (5, 10);
                END
            ");

            // 🛠️ 3. Create Stored Procedures
            // USP for Top Selling Books All Time
            await context.Database.ExecuteSqlRawAsync(@"
                IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Usp_GetTopNSellingBooksAllTime]') AND type in (N'P', N'PC'))
                BEGIN
                    EXEC('CREATE PROCEDURE [dbo].[Usp_GetTopNSellingBooksAllTime]
                    AS
                    BEGIN
                        SET NOCOUNT ON;
                        SELECT TOP 5 b.Id as BookId, b.BookName, b.AuthorName, b.Image, ISNULL(SUM(od.Quantity), 0) as TotalUnitSold
                        FROM Book b
                        LEFT JOIN OrderDetail od ON b.Id = od.BookId
                        LEFT JOIN [Order] o ON o.Id = od.OrderId
                        GROUP BY b.Id, b.BookName, b.AuthorName, b.Image
                        ORDER BY TotalUnitSold DESC
                    END')
                END");

            // USP for Top Selling Books By Date
            await context.Database.ExecuteSqlRawAsync(@"
                IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Usp_GetTopNSellingBooksByDate]') AND type in (N'P', N'PC'))
                BEGIN
                    EXEC('CREATE PROCEDURE [dbo].[Usp_GetTopNSellingBooksByDate]
                        @startDate datetime,
                        @endDate datetime
                    AS
                    BEGIN
                        SET NOCOUNT ON;
                        SELECT TOP 5 b.Id as BookId, b.BookName, b.AuthorName, b.Image, ISNULL(SUM(od.Quantity), 0) as TotalUnitSold
                        FROM Book b
                        LEFT JOIN OrderDetail od ON b.Id = od.BookId
                        LEFT JOIN [Order] o ON o.Id = od.OrderId
                        WHERE o.CreateDate >= @startDate AND o.CreateDate <= @endDate
                        GROUP BY b.Id, b.BookName, b.AuthorName, b.Image
                        ORDER BY TotalUnitSold DESC
                    END')
                END");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error seeding stored procedures and tables: " + ex.Message);
        }
    }

    #region private methods

    private static async Task SeedGenreAsync(ApplicationDbContext context)
    {
        var genres = new[]
         {
            new Genre { GenreName = "Romance" },
            new Genre { GenreName = "Action" },
            new Genre { GenreName = "Thriller" },
            new Genre { GenreName = "Crime" },
            new Genre { GenreName = "SelfHelp" },
            new Genre { GenreName = "Programming" }
        };

        await context.Genres.AddRangeAsync(genres);
        await context.SaveChangesAsync();
    }

    private static async Task SeedOrderStatusAsync(ApplicationDbContext context)
    {
        var orderStatuses = new[]
        {
            new OrderStatus { StatusId = 1, StatusName = "Pending" },
            new OrderStatus { StatusId = 2, StatusName = "Shipped" },
            new OrderStatus { StatusId = 3, StatusName = "Delivered" },
            new OrderStatus { StatusId = 4, StatusName = "Cancelled" },
            new OrderStatus { StatusId = 5, StatusName = "Returned" },
            new OrderStatus { StatusId = 6, StatusName = "Refund" }
        };

        await context.orderStatuses.AddRangeAsync(orderStatuses);
        await context.SaveChangesAsync();
    }

    private static async Task UpdateMissingDescriptionsAsync(ApplicationDbContext context)
    {
        var booksToUpdate = await context.Books.Where(b => string.IsNullOrEmpty(b.Description)).ToListAsync();
        if (booksToUpdate.Any())
        {
            var seedData = new List<Book>
            {
                new Book { BookName = "Pride and Prejudice", Description = "Bản tình ca kinh điển về tình yêu và định kiến xã hội thời Nhiếp chính tại Anh." },
                new Book { BookName = "The Notebook", Description = "Câu chuyện tình yêu đầy cảm động vượt qua thời gian và những thử thách khắc nghiệt." },
                new Book { BookName = "Outlander", Description = "Hành trình xuyên không đầy phiêu lưu và lãng mạn giữa thế kỷ 20 và vùng cao nguyên Scotland thế kỷ 18." },
                new Book { BookName = "Me Before You", Description = "Câu chuyện về Louisa Clark và Will Traynor, một bài học sâu sắc về sự lựa chọn và ý nghĩa cuộc sống." },
                new Book { BookName = "The Fault in Our Stars", Description = "Hai người trẻ với căn bệnh ung thư cùng nhau khám phá ý nghĩa của cuộc sống và tình yêu." },
                new Book { BookName = "The Bourne Identity", Description = "Hành trình đi tìm danh tính của một điệp viên bị mất trí nhớ giữa lằn ranh sinh tử." },
                new Book { BookName = "Die Hard", Description = "Cuộc chiến nghẹt thở của một cảnh sát chống lại nhóm khủng bố trong một tòa nhà chọc trời." },
                new Book { BookName = "Jurassic Park", Description = "Sự hồi sinh của những sinh vật tiền sử và tham vọng khủng khiếp của con người." },
                new Book { BookName = "The Da Vinci Code", Description = "Giải mã những mật mã ẩn giấu trong các tác phẩm nghệ thuật để tìm ra chân lý lịch sử." },
                new Book { BookName = "The Hunger Games", Description = "Cuộc chiến sinh tồn khốc liệt giữa những thanh thiếu niên trong một xã hội tương lai tàn khốc." },
                new Book { BookName = "Gone Girl", Description = "Sự mất tích bí ẩn và những góc khuất đen tối trong hôn nhân của một cặp vợ chồng." },
                new Book { BookName = "The Girl with the Dragon Tattoo", Description = "Điều tra về vụ mất tích bí ẩn diễn ra trong một gia tộc quyền lực từ nhiều thập kỷ trước." },
                new Book { BookName = "The Silence of the Lambs", Description = "Cuộc đấu trí giữa một nữ học viên FBI và kẻ sát nhân hàng loạt khét tiếng nhất lịch sử văn học." },
                new Book { BookName = "Before I Go to Sleep", Description = "Mỗi buổi sáng thức dậy là một ngày mới hoàn toàn với một người phụ nữ bị mất trí nhớ ngắn hạn." },
                new Book { BookName = "The Girl on the Train", Description = "Chứng kiến một sự việc bất ngờ qua cửa sổ tàu hỏa kéo theo một chuỗi bi kịch khó lường." },
                new Book { BookName = "The Godfather", Description = "Đế chế tội phạm gia đình Corleone và những quy luật ngầm trong giới mafia." },
                new Book { BookName = "The Girl with the Dragon Tattoo 2", Description = "Phần tiếp theo đầy kịch tính về Lisbeth Salander và hành trình chống lại bóng tối xã hội." },
                new Book { BookName = "The Cuckoo's Calling", Description = "Thám tử tư Cormoran Strike điều tra cái chết bí ẩn của một người mẫu nổi tiếng." },
                new Book { BookName = "In Cold Blood", Description = "Phóng sự tội phạm dựa trên sự kiện có thật đầy ám ảnh về một vụ thảm sát kinh hoàng." },
                new Book { BookName = "The Silence of the Lambs 2", Description = "Sự trở lại của Dr. Hannibal Lecter trong một cuộc rượt đuổi nghẹt thở xuyên lục địa." },
                new Book { BookName = "The 7 Habits of Highly Effective People", Description = "Kỹ năng sống và làm việc hiệu quả dựa trên những nguyên lý cốt lõi của tính cách." },
                new Book { BookName = "How to Win Friends and Influence People", Description = "Nghệ thuật giao tiếp và thu phục lòng người mang lại thành công trong cuộc sống và công việc." },
                new Book { BookName = "Atomic Habits", Description = "Thay đổi thói quen nhỏ mỗi ngày để đạt được những thành tựu to lớn trong tương lai." },
                new Book { BookName = "The Subtle Art of Not Giving a F*ck", Description = "Một cách tiếp cận thực tế và thẳng thắn để sống một cuộc đời có ý nghĩa hơn." },
                new Book { BookName = "You Are a Badass", Description = "Đánh thức sức mạnh tiềm ẩn bên trong bạn để tạo ra một cuộc đời bạn hằng mong ước." },
                new Book { BookName = "Clean Code", Description = "Cẩm nang về kỹ thuật lập trình mã sạch để tạo ra những phần mềm dễ bảo trì và mở rộng." },
                new Book { BookName = "Design Patterns", Description = "Giới thiệu các kiến trúc mẫu tối ưu để giải quyết các vấn đề phổ biến trong thiết kế phần mềm." },
                new Book { BookName = "Code Complete", Description = "Hướng dẫn chi tiết và thực chất về chu trình xây dựng và phát triển phần mềm chất lượng cao." },
                new Book { BookName = "The Pragmatic Programmer", Description = "Những lời khuyên và kỹ năng thực tiễn để trở thành một lập trình viên chuyên nghiệp và sáng tạo." },
                new Book { BookName = "Head First Design Patterns", Description = "Học về các mẫu thiết kế thông qua hình ảnh sinh động và các bài tập tình huống thực tế." }
            };

            foreach (var book in booksToUpdate)
            {
                var data = seedData.FirstOrDefault(s => s.BookName == book.BookName);
                if (data != null)
                {
                    book.Description = data.Description;
                }
            }
            await context.SaveChangesAsync();
        }
    }

    private static async Task SeedBooksAsync(ApplicationDbContext context)
    {
        var books = new List<Book>
        {
            // Romance Books (GenreId = 1)
            new Book { BookName = "Pride and Prejudice", AuthorName = "Jane Austen", Price = 12.99, GenreId = 1, Image = "010d6301-a7f7-4c72-9f5d-9fa56fca46fa.jpg", Description = "Bản tình ca kinh điển về tình yêu và định kiến xã hội thời Nhiếp chính tại Anh." },
            new Book { BookName = "The Notebook", AuthorName = "Nicholas Sparks", Price = 11.99, GenreId = 1, Image = "03fa02fd-1795-44ff-98db-d5a98a93b232.jpg", Description = "Câu chuyện tình yêu đầy cảm động vượt qua thời gian và những thử thách khắc nghiệt." },
            new Book { BookName = "Outlander", AuthorName = "Diana Gabaldon", Price = 14.99, GenreId = 1, Image = "04da70bd-bab9-4c81-8adf-1f0598fa68a7.jpg", Description = "Hành trình xuyên không đầy phiêu lưu và lãng mạn giữa thế kỷ 20 và vùng cao nguyên Scotland thế kỷ 18." },
            new Book { BookName = "Me Before You", AuthorName = "Jojo Moyes", Price = 10.99, GenreId = 1, Image = "055f5254-c70d-4586-afdf-1d44653319c8.jpg", Description = "Câu chuyện về Louisa Clark và Will Traynor, một bài học sâu sắc về sự lựa chọn và ý nghĩa cuộc sống." },
            new Book { BookName = "The Fault in Our Stars", AuthorName = "John Green", Price = 9.99, GenreId = 1, Image = "0d16d1f0-06b1-463f-a89b-87fd0b1c0349.jpg", Description = "Hai người trẻ với căn bệnh ung thư cùng nhau khám phá ý nghĩa của cuộc sống và tình yêu." },
            
            // Action Books (GenreId = 2)
            new Book { BookName = "The Bourne Identity", AuthorName = "Robert Ludlum", Price = 14.99, GenreId = 2, Image = "0ee05c98-d599-4fe3-aaa5-66a33388bb6e.jpg", Description = "Hành trình đi tìm danh tính của một điệp viên bị mất trí nhớ giữa lằn ranh sinh tử." },
            new Book { BookName = "Die Hard", AuthorName = "Roderick Thorp", Price = 13.99, GenreId = 2, Image = "19a0424c-f7a4-491c-a1a8-3222c1ccc6ee.jpg", Description = "Cuộc chiến nghẹt thở của một cảnh sát chống lại nhóm khủng bố trong một tòa nhà chọc trời." },
            new Book { BookName = "Jurassic Park", AuthorName = "Michael Crichton", Price = 15.99, GenreId = 2, Image = "23bcdfa1-9559-464f-892a-6e8abfe9c4b2.jpg", Description = "Sự hồi sinh của những sinh vật tiền sử và tham vọng khủng khiếp của con người." },
            new Book { BookName = "The Da Vinci Code", AuthorName = "Dan Brown", Price = 12.99, GenreId = 2, Image = "30a5dacf-c3f6-470a-a798-eecb81337f5c.jpg", Description = "Giải mã những mật mã ẩn giấu trong các tác phẩm nghệ thuật để tìm ra chân lý lịch sử." },
            new Book { BookName = "The Hunger Games", AuthorName = "Suzanne Collins", Price = 11.99, GenreId = 2, Image = "316e3021-27ec-4a28-aa76-0be563c2342e.jpg", Description = "Cuộc chiến sinh tồn khốc liệt giữa những thanh thiếu niên trong một xã hội tương lai tàn khốc." },
            
            // Thriller Books (GenreId = 3)
            new Book { BookName = "Gone Girl", AuthorName = "Gillian Flynn", Price = 11.99, GenreId = 3, Image = "41fd5045-799c-4603-ab70-24caa4e4dd5b.jpg", Description = "Sự mất tích bí ẩn và những góc khuất đen tối trong hôn nhân của một cặp vợ chồng." },
            new Book { BookName = "The Girl with the Dragon Tattoo", AuthorName = "Stieg Larsson", Price = 10.99, GenreId = 3, Image = "4444817d-1651-40f6-8057-7e0e0aed0a88.jpg", Description = "Điều tra về vụ mất tích bí ẩn diễn ra trong một gia tộc quyền lực từ nhiều thập kỷ trước." },
            new Book { BookName = "The Silence of the Lambs", AuthorName = "Thomas Harris", Price = 12.99, GenreId = 3, Image = "47ac54d8-35ad-47af-8efb-ffcebed81d3b.jpg", Description = "Cuộc đấu trí giữa một nữ học viên FBI và kẻ sát nhân hàng loạt khét tiếng nhất lịch sử văn học." },
            new Book { BookName = "Before I Go to Sleep", AuthorName = "S.J. Watson", Price = 9.99, GenreId = 3, Image = "593f423f-aa6d-4d7e-aa87-703e37e8e3b2.jpg", Description = "Mỗi buổi sáng thức dậy là một ngày mới hoàn toàn với một người phụ nữ bị mất trí nhớ ngắn hạn." },
            new Book { BookName = "The Girl on the Train", AuthorName = "Paula Hawkins", Price = 13.99, GenreId = 3, Image = "5a4bfb1e-f654-4f61-abb0-69941b86494a.jpg", Description = "Chứng kiến một sự việc bất ngờ qua cửa sổ tàu hỏa kéo theo một chuỗi bi kịch khó lường." },
            
            // Crime Books (GenreId = 4)
            new Book { BookName = "The Godfather", AuthorName = "Mario Puzo", Price = 13.99, GenreId = 4, Image = "5be3022a-8371-4524-86e3-bee4711f872f.jpg", Description = "Đế chế tội phạm gia đình Corleone và những quy luật ngầm trong giới mafia." },
            new Book { BookName = "The Girl with the Dragon Tattoo 2", AuthorName = "Stieg Larsson", Price = 12.99, GenreId = 4, Image = "5f278d11-f44a-4827-9a0c-f77a5d6aebad.jpg", Description = "Phần tiếp theo đầy kịch tính về Lisbeth Salander và hành trình chống lại bóng tối xã hội." },
            new Book { BookName = "The Cuckoo's Calling", AuthorName = "Robert Galbraith", Price = 14.99, GenreId = 4, Image = "61322b9e-e2f7-4ade-bbc3-275357b43543.jpg", Description = "Thám tử tư Cormoran Strike điều tra cái chết bí ẩn của một người mẫu nổi tiếng." },
            new Book { BookName = "In Cold Blood", AuthorName = "Truman Capote", Price = 11.99, GenreId = 4, Image = "61ca2a0d-b1b5-45ed-8a91-b71252066f4c.jpg", Description = "Phóng sự tội phạm dựa trên sự kiện có thật đầy ám ảnh về một vụ thảm sát kinh hoàng." },
            new Book { BookName = "The Silence of the Lambs 2", AuthorName = "Thomas Harris", Price = 15.99, GenreId = 4, Image = "667a77f7-0ed9-457d-b65e-9981b9556c5f.jpg", Description = "Sự trở lại của Dr. Hannibal Lecter trong một cuộc rượt đuổi nghẹt thở xuyên lục địa." },
            
            // SelfHelp Books (GenreId = 5)
            new Book { BookName = "The 7 Habits of Highly Effective People", AuthorName = "Stephen R. Covey", Price = 9.99, GenreId = 5, Image = "69641a9e-f9f7-4a6f-a674-2e33f24197ca.jpg", Description = "Kỹ năng sống và làm việc hiệu quả dựa trên những nguyên lý cốt lõi của tính cách." },
            new Book { BookName = "How to Win Friends and Influence People", AuthorName = "Dale Carnegie", Price = 8.99, GenreId = 5, Image = "6c07fd21-281a-478f-9190-209f41b03067.jpg", Description = "Nghệ thuật giao tiếp và thu phục lòng người mang lại thành công trong cuộc sống và công việc." },
            new Book { BookName = "Atomic Habits", AuthorName = "James Clear", Price = 10.99, GenreId = 5, Image = "73c25d6f-4b74-41d5-a6bd-7fa6e601c3bb.jpg", Description = "Thay đổi thói quen nhỏ mỗi ngày để đạt được những thành tựu to lớn trong tương lai." },
            new Book { BookName = "The Subtle Art of Not Giving a F*ck", AuthorName = "Mark Manson", Price = 7.99, GenreId = 5, Image = "8287adbe-7832-4d3a-9367-cfa7f6fe767f.jpg", Description = "Một cách tiếp cận thực tế và thẳng thắn để sống một cuộc đời có ý nghĩa hơn." },
            new Book { BookName = "You Are a Badass", AuthorName = "Jen Sincero", Price = 11.99, GenreId = 5, Image = "8c855d0f-cfbf-4701-8c8f-e00921326376.jpg", Description = "Đánh thức sức mạnh tiềm ẩn bên trong bạn để tạo ra một cuộc đời bạn hằng mong ước." },
            
            // Programming Books (GenreId = 6)
            new Book { BookName = "Clean Code", AuthorName = "Robert C. Martin", Price = 19.99, GenreId = 6, Image = "8dce0de9-0c40-4ef1-bc7d-e71d78aed3b9.jpg", Description = "Cẩm nang về kỹ thuật lập trình mã sạch để tạo ra những phần mềm dễ bảo trì và mở rộng." },
            new Book { BookName = "Design Patterns", AuthorName = "Erich Gamma", Price = 17.99, GenreId = 6, Image = "95d004ad-33b6-41c4-aa87-c9fa9104d41c.jpg", Description = "Giới thiệu các kiến trúc mẫu tối ưu để giải quyết các vấn đề phổ biến trong thiết kế phần mềm." },
            new Book { BookName = "Code Complete", AuthorName = "Steve McConnell", Price = 21.99, GenreId = 6, Image = "9e1bd964-656d-47b3-8829-648dcdba0eb1.jpg", Description = "Hướng dẫn chi tiết và thực chất về chu trình xây dựng và phát triển phần mềm chất lượng cao." },
            new Book { BookName = "The Pragmatic Programmer", AuthorName = "Andrew Hunt", Price = 18.99, GenreId = 6, Image = "9fc8e80a-21f1-4d9d-8d46-868ea952c837.jpg", Description = "Những lời khuyên và kỹ năng thực tiễn để trở thành một lập trình viên chuyên nghiệp và sáng tạo." },
            new Book { BookName = "Head First Design Patterns", AuthorName = "Eric Freeman", Price = 20.99, GenreId = 6, Image = "a0b3b6fd-eef3-40b8-88b7-2ff2bf761d9f.jpg", Description = "Học về các mẫu thiết kế thông qua hình ảnh sinh động và các bài tập tình huống thực tế." }
        };

        await context.Books.AddRangeAsync(books);
        await context.SaveChangesAsync();
    }

    #endregion
}
