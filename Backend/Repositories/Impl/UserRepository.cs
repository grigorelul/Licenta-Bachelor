﻿using Microsoft.EntityFrameworkCore;
using Models;

namespace Repositories;

public class UserRepository : IUserRepository
{
    private readonly MyDbContext _context;
    private readonly DbSet<User> _users;

    public UserRepository(MyDbContext context)
    {
        _context = context;
        _users = context.Users;
    }
    public async Task<IEnumerable<User>> GetUsersAsync()
    {
        return await _users.ToListAsync();
    }

    public async Task<IEnumerable<User>> GetUserByNumeAsync(string username)
    {
        return await _users.Where(u => u.Nume == username).ToListAsync();
    }
    
    public async Task<User> CreateUserAsync(User user)
    {
        await _users.AddAsync(user);
        await _context.SaveChangesAsync();
        return user;
    }

    public async Task<User> GetUserAsync(Guid id)
    {
        return await _users.FindAsync(id);
    }

    public async Task<User> DeleteUserAsync(Guid id)
    {
        var user = await _users.FindAsync(id);
        if (user == null)
        {
            return null;
        }

        _users.Remove(user);
        await _context.SaveChangesAsync();
        return user;
    }

    public async Task<User> GetUserByEmailAsync(string email)
    {
        return await _users.FirstOrDefaultAsync(u => u.Email == email);
    }

    public async Task<User> UpdateUserAsync(User user)
    {
        _users.Update(user);
        await _context.SaveChangesAsync();
        return user;
    }

    public async Task<IEnumerable<Attendance>> GetUserAttendencesAsync(Guid id)
    {
        return await _context.Attendances.Where(a => a.UserId == id).ToListAsync();
    }
   

}
