using TodoApi.DTOs;

namespace TodoApi.Models
{
    public class Manager
    {
        public Guid Id { get; set; }
        public string Nume { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Parola { get; set; } = string.Empty; 
        public bool? Admin { get; set; }

        public List<User> Users { get; set; } = new List<User>();
        
        public List<Attendance> Attendances { get; set; } = new List<Attendance>();

        public static Manager FromManagerDto(ManagerDto managerDto) =>
            new()
            {
                Id = managerDto.Id,
                Nume = managerDto.Nume,
                Email = managerDto.Email
            };
    }
}
