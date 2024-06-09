using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TodoApi.Migrations
{
    /// <inheritdoc />
    public partial class migration2 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<Guid>(
                name: "ManagerId",
                table: "Attendances",
                type: "uniqueidentifier",
                nullable: false,
                defaultValue: new Guid("00000000-0000-0000-0000-000000000000"));

            migrationBuilder.CreateIndex(
                name: "IX_Attendances_ManagerId",
                table: "Attendances",
                column: "ManagerId");

            migrationBuilder.AddForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances",
                column: "ManagerId",
                principalTable: "Managers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Attendances_Managers_ManagerId",
                table: "Attendances");

            migrationBuilder.DropIndex(
                name: "IX_Attendances_ManagerId",
                table: "Attendances");

            migrationBuilder.DropColumn(
                name: "ManagerId",
                table: "Attendances");
        }
    }
}
