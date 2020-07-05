def get_sample(model, dataset_loader, device, size, prime_text, top_k=5):
    model.eval()
    model.to(device)
    output_text = [character for character in prime_text]
    previous_hidden_states = model.init_hidden_states(1)
    for character in prime_text:
        predicted_character, previous_hidden_states = model.predict(
            character, dataset_loader, device, previous_hidden_states, top_k=top_k
        )
    output_text.append(predicted_character)
    for _ in range(size):
        predicted_character, previous_hidden_states = model.predict(
            output_text[-1], dataset_loader, device, previous_hidden_states, top_k=top_k
        )
        output_text.append(predicted_character)
    return "".join(output_text)
