namespace BookShoppingCartMvcUI.Models.Recommendation
{
    public class AlgorithmParametersDto
    {
        public string Model { get; set; }   // mf, usercf, itemcf
        public int Top_N { get; set; }
    }
}
