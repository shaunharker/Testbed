import torch

default_prompt = """
To Dr. Faustus in his study Mephistopheles told the history of the Creation, saying:
"The endless praises of the choirs of angels had begun to grow wearisome; for, after all, did he not deserve their praise? Had he not given them endless joy? Would it not be more amusing to obtain undeserved praise, to be worshipped by beings whom he tortured? He smiled inwardly, and resolved that the great drama should be performed.

"For countless ages the hot nebula whirled aimlessly through space. At length it began to take shape, the central mass threw off planets, the planets cooled, boiling seas and burning mountains heaved and tossed, from black masses of cloud hot sheets of rain deluged the barely solid crust. And now the first germ of life grew in the depths of the ocean, and developed rapidly in the fructifying warmth into vast forest trees, huge ferns springing from the damp mould, sea monsters breeding, fighting, devouring, and passing away. And from the monsters, as the play unfolded itself, Man was born, with the power of thought, the knowledge of good and evil, and the cruel thirst for worship. And Man saw that all is passing in this mad, monstrous world, that all is struggling to snatch, at any cost, a few brief moments of life before Death's inexorable decree. And Man said: `There is a hidden purpose, could we but fathom it, and the purpose is good; for we must reverence something, and in the visible world there is nothing worthy of reverence.' And Man stood aside from the struggle, resolving that God intended harmony to come out of chaos by human efforts. And when he followed the instincts which God had transmitted to him from his ancestry of beasts of prey, he called it Sin, and asked God to forgive him. But he doubted whether he could be justly forgiven, until he invented a divine Plan by which God's wrath was to have been appeased. And seeing the present was bad, he made it yet worse, that thereby the future might be better. And he gave God thanks for the strength that enabled him to forgo even the joys that were possible. And God smiled; and when he saw that Man had become perfect in renunciation and worship, he sent another sun through the sky, which crashed into Man's sun; and all returned again to nebula.

"`Yes,' he murmured, `it was a good play; I will have it performed again.'"

Such, in outline, but even more purposeless, more void of meaning, is the world which Science presents for our belief. Amid such a world, if anywhere, our ideals henceforward must find a home. That Man is the product of causes which had no prevision of the end they were achieving; that his origin, his growth, his hopes and fears, his loves and his beliefs, are but the outcome of accidental collocations of atoms; that no fire, no heroism, no intensity of thought and feeling, can preserve an individual life beyond the grave; that all the labours of the ages, all the devotion, all the inspiration, all the noonday brightness of human genius, are destined to extinction in the vast death of the solar system, and that the whole temple of Man's achievement must inevitably be buried beneath the debris of a universe in ruins--all these things, if not quite beyond dispute, are yet so nearly certain, that no philosophy which rejects them can hope to stand. Only within the scaffolding of these truths, only on the firm foundation of unyielding despair, can the soul's habitation henceforth be safely built.

How, in such an alien and inhuman world, can so powerless a creature as Man preserve his aspirations untarnished? A strange mystery it is that Nature, omnipotent but blind, in the revolutions of her secular hurryings through the abysses of space, has brought forth at last a child, subject still to her power, but gifted with sight, with knowledge of good and evil, with the capacity of judging all the works of his unthinking Mother. In spite of Death, the mark and seal of the parental control, Man is yet free, during his brief years, to examine, to criticise, to know, and in imagination to create. To him alone, in the world with which he is acquainted, this freedom belongs; and in this lies his superiority to the resistless forces that control his outward life.

The savage, like ourselves, feels the oppression of his impotence before the powers of Nature; but having in himself nothing that he respects more than Power, he is willing to prostrate himself before his gods, without inquiring whether they are worthy of his worship. Pathetic and very terrible is the long history of cruelty and torture, of degradation and human sacrifice, endured in the hope of placating the jealous gods: surely, the trembling believer thinks, when what is most precious has been freely given, their lust for blood must be appeased, and more will not be required. The religion of Moloch--as such creeds may be generically called--is in essence the cringing submission of the slave, who dare not, even in his heart, allow the thought that his master deserves no adulation. Since the independence of ideals is not yet acknowledged, Power may be freely worshipped, and receive an unlimited respect, despite its wanton infliction of pain.

But gradually, as morality grows bolder, the claim of the ideal world begins to be felt; and worship, if it is not to cease, must be given to gods of another kind than those created by the savage. Some, though they feel the demands of the ideal, will still consciously reject them, still urging that naked Power is worthy of worship. Such is the attitude inculcated in God's answer to Job out of the whirlwind: the divine power and knowledge are paraded, but of the divine goodness there is no hint. Such also is the attitude of those who, in our own day, base their morality upon the struggle for survival, maintaining that the survivors are necessarily the fittest. But others, not content with an answer so repugnant to the moral sense, will adopt the position which we have become accustomed to regard as specially religious, maintaining that, in some hidden manner, the world of fact is really harmonious with the world of ideals. Thus Man creates God, all-powerful and all-good, the mystic unity of what is and what should be.

But the world of fact, after all, is not good; and, in submitting our judgment to it, there is an element of slavishness from which our thoughts must be purged. For in all things it is well to exalt the dignity of Man, by freeing him as far as possible from the tyranny of non-human Power. When we have realised that Power is largely bad, that man, with his knowledge of good and evil, is but a helpless atom in a world which has no such knowledge, the choice is again presented to us: Shall we worship Force, or shall we worship Goodness? Shall our God exist and be evil, or shall he be recognised as the creation of our own conscience?

The answer to this question is very momentous, and affects profoundly our whole morality. The worship of Force, to which Carlyle and Nietzsche and the creed of Militarism have accustomed us, is the result of failure to maintain our own ideals against a hostile universe: it is itself a prostrate submission to evil, a sacrifice of our best to Moloch. If strength indeed is to be respected, let us respect rather the strength of those who refuse that false "recognition of facts" which fails to recognise that facts are often bad. Let us admit that, in the world we know, there are many things that would be better otherwise, and that the ideals to which we do and must adhere are not realised in the realm of matter. Let us preserve our respect for truth, for beauty, for the ideal of perfection which life does not permit us to attain, though none of these things meet with the approval of the unconscious universe. If Power is bad, as it seems to be, let us reject it from our hearts. In this lies Man's true freedom: in determination to worship only the God created by our own love of the good, to respect only the heaven which inspires the insight of our best moments. In action, in desire, we must submit perpetually to the tyranny of outside forces; but in thought, in aspiration, we are free, free from our fellow-men, free from the petty planet on which our bodies impotently crawl, free even, while we live, from the tyranny of death. Let us learn, then, that energy of faith which enables us to live constantly in the vision of the good; and let us descend, in action, into the world of fact, with that vision always before us.

When first the opposition of fact and ideal grows fully visible, a spirit of fiery revolt, of fierce hatred of the gods, seems necessary to the assertion of freedom. To defy with Promethean constancy a hostile universe, to keep its evil always in view, always actively hated, to refuse no pain that the malice of Power can invent, appears to be the duty of all who will not bow before the inevitable. But indignation is still a bondage, for it compels our thoughts to be occupied with an evil world; and in the fierceness of desire from which rebellion springs there is a kind of self-assertion which it is necessary for the wise to overcome. Indignation is a submission of our thoughts, but not of our desires; the Stoic freedom in which wisdom consists is found in the submission of our desires, but not of our thoughts. From the submission of our desires springs the virtue of resignation; from the freedom of our thoughts springs the whole world of art and philosophy, and the vision of beauty by which, at last, we half reconquer the reluctant world."""

def autocomplete(model, encode, decode, prompt=None, n_ctx=None, temp=1.0, n_generate=512, device=None, verbose=False):
    """
    Autocomplete using the model

    ## Args
    * `prompt: str` an optional prompt to begin with. Defaults to a prefix of Bertrand Russell's A Free Man's Worship
    * `n_ctx: int` the number of bytes/tokens in the context window
    * `encode` the function that can turn an str into a sequence of bytes/tokens suitable for the model.
    defaults to utf8encode
    * `decode` the function that can turn the sequences of bytes/tokens used by the model to a str
    defaults to utf8decode

    """
    Categorical = torch.distributions.Categorical
    if n_ctx is None:
        n_ctx = model.n_ctx
    if prompt is None:
        prompt = default_prompt
    if device is None:
        device = model.device
    x = encode(prompt)
    x = x[-n_ctx:]
    prompt = decode(x)
    if verbose:
        print(f"=== Prompt ===\n{prompt}\n=== Autocompletion ===\n")
    def sampler(x):
        x = list(x)
        for _ in range(n_generate):
            probs = model.inference(torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)).view(-1)[-model.n_vocab_out:]
            if temp > 0:
                y = Categorical(probs=probs**(1.0/temp)).sample().item()
            else:
                y = torch.argmax(probs).item()
            x = (x + [y])[-n_ctx:]
            yield y
    return decode(list(sampler(x)))
