
namespace Models;
public class ManagerDto
{
    public Guid Id { get; set; }
    public string Nume { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    
    public static ManagerDto  FromManagerToManagerDto(Manager manager) =>
        new()
        {
            Id = manager.Id,
            Nume = manager.Nume,
            Email = manager.Email
        };
}