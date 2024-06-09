using System;
using TodoApi.Models;
namespace TodoApi.DTOs;


public class UserDto
{
    public Guid Id { get; set; }
    public string Nume { get; set; } = string.Empty;
    public string Email { get; set; } = string.Empty;

    public static UserDto FromUser(User user) =>
        new ()
    {
        Id = user.Id,
        Nume = user.Nume,
        Email = user.Email
    };
}
