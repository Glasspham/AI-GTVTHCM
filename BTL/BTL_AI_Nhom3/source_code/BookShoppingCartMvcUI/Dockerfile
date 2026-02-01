FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY ["BookShoppingCartMvcUI.csproj", "./"]
RUN dotnet restore "./BookShoppingCartMvcUI.csproj"

# Copy everything else and build
COPY . .
RUN dotnet publish "BookShoppingCartMvcUI.csproj" -c Release -o /app/publish

# Run stage
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "BookShoppingCartMvcUI.dll"]
