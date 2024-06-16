
namespace Models;
    public class User
    {
        public Guid Id { get; set; }
        public string Nume { get; set; } = string.Empty;
        public string Email { get; set; } = string.Empty;
        public string Parola { get; set; } = string.Empty;
        public bool? Admin { get; set; }
        
        // Adminul vede toate pontările 
        //Admin = null => User
        //Admin = true => Admin
        //Admin = false => Manager
        
        // Relații
        public List<Manager> Managers { get; set;} = new List<Manager>();
        // pontările unui angajat
        public List<Attendance> Attendances { get; set; } = new List<Attendance>();


        public static User FromUserDtoToUser(UserDto userDto) =>
            new()
            {
                Id = userDto.Id,
                Nume = userDto.Nume,
                Email = userDto.Email
            };


    }
