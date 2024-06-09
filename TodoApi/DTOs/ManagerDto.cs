using TodoApi.Models;

namespace TodoApi.DTOs;
public class ManagerDto
{
    public Guid Id { get; set; }
    public string Nume { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;
    
    public static ManagerDto FromManager(Manager manager) =>
        new()
        {
            Id = manager.Id,
            Nume = manager.Nume,
            Email = manager.Email
        };
}