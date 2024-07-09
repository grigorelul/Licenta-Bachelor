
namespace Models;
    public class Attendance
    {
        public Guid Id { get; set; } // Cheie primară

        public DateTime DataSosire { get; set; } // Coloană datetime
        public DateTime? DataPlecare { get; set; } // Coloană datetime
        public Guid? UserId { get; set; } // Cheie străină către User
        public Guid? ManagerId { get; set; } // Cheie străină către Manager

        public User? User { get; set; }
        
        public Manager? Manager { get; set; }
        
        public static Attendance FromAttendanceDtoToAttendance(AttendanceDto attendanceDto) =>
            new()
            {
                Id = attendanceDto.Id,
                DataSosire = attendanceDto.DataSosire,
                DataPlecare = attendanceDto.DataPlecare,
            };
    }