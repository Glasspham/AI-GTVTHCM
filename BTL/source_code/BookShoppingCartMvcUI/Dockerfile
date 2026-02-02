FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY ["BookShoppingCartMvcUI.csproj", "./"]
RUN dotnet restore "./BookShoppingCartMvcUI.csproj"

# Copy everything else and build
COPY . .

# 🛠️ PATCH Program.cs for Docker (HTTP Environment)
RUN sed -i 's/CookieSecurePolicy.Always/CookieSecurePolicy.SameAsRequest/g' Program.cs && \
    sed -i 's/SameSiteMode.None/SameSiteMode.Lax/g' Program.cs && \
    sed -i 's/options.SignIn.RequireConfirmedAccount = true/options.SignIn.RequireConfirmedAccount = false/g' Program.cs && \
    sed -i 's/app.UseHttpsRedirection();//g' Program.cs && \
    sed -i 's/app.UseHsts();//g' Program.cs

# 🖼️ FIX Case Sensitivity for images in Linux
RUN sed -i 's/banner2.jpg/banner2.JPG/g' Views/Main/Index.cshtml

# 🤖 CONNECT to Python API via Docker Network (Use service name 'api' instead of '127.0.0.1')
RUN sed -i 's/127.0.0.1:8000/api:8000/g' Services/RecommendationService.cs

RUN dotnet publish "BookShoppingCartMvcUI.csproj" -c Release -o /app/publish

# Run stage
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "BookShoppingCartMvcUI.dll"]
