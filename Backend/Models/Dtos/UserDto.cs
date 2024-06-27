
namespace Models;

public class UserDto
{
    public Guid Id { get; set; }
    public string Nume { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    public string Parola { get; set; } = string.Empty;


    public static UserDto FromUserToUserDto(User user) =>
        new ()
    {
        Id = user.Id,
        Nume = user.Nume,
        Email = user.Email,
        Parola = user.Parola

    };
}
