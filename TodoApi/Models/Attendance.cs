namespace TodoApi.Models
{
    public class Attendance
    {
        public Guid Id { get; set; } // Cheie primară

        public DateTime DataPosire { get; set; } // Coloană datetime
        public DateTime DataPlecare { get; set; } // Coloană datetime
        public Guid UserId { get; set; } // Cheie străină către User
        public Guid ManagerId { get; set; } // Cheie străină către Manager

        public User? User { get; set; }
        
        public Manager? Manager { get; set; }
    }
}